"""Contract and rendering regressions for corpus-based president profiles."""

from __future__ import annotations

import html as html_lib
import json
import inspect
import math
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import (
    ai_labels,
    indices,
    llm_annotations,
    profiles,
    profiles_site,
    topic_quality,
)
from presidential_profiles.corpus import PARTY


BASE_ISSUES = ["Economic policy", "Foreign affairs", "Civil rights & race"]
TOPICS = [
    "Economic stewardship",
    "International relations",
    "Rights and citizenship",
    "Government operations",
    "National identity",
    "Public welfare",
]
SMALL_PRESIDENTS = [
    "William Harrison",
    "James A. Garfield",
    "Franklin D. Roosevelt",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ai_profile(position: int, n_speeches: int, thin: bool) -> dict:
    top_topics = [
        {
            "name": name,
            "domain": f"Domain {index + 1}",
            "definition": f"Frozen definition {index + 1}.",
            "share": round(22.0 - index * 2.2 + position / 100, 2),
            "rel": round(3.0 - index * 0.8, 2),
            "n": 24 - index,
        }
        for index, name in enumerate(TOPICS)
    ]
    flags = {"party_attack": 4.0, "enemy_naming": 3.0, "zero_sum": 2.0}
    proposal_values = {
        "proposal": 36.0,
        "mixed": 14.0,
        "values": 28.0,
        "neither": 22.0,
    }
    absolutes = {
        **flags,
        "proposal": proposal_values["proposal"],
        "values": proposal_values["values"],
        "topic_breadth": 4.2,
    }
    # Seed thin records with an invalid synthetic midpoint. The normalized
    # model must withdraw it everywhere instead of allowing it to escape.
    ai_radar = {
        spec["key"]: {
            "absolute": absolutes[spec["key"]],
            "percentile": 50.0 if thin else float(40 + position % 45),
        }
        for spec in profiles_site.AI_MEASURE_SPECS
    }
    return {
        "n_paragraphs": 120 + position,
        "n_speeches": n_speeches,
        "low_confidence": thin,
        "top_topics": top_topics,
        "distinctive_topics": top_topics[:3],
        "domains": [
            {"name": item["domain"], "share": item["share"], "rel": item["rel"]}
            for item in top_topics[:5]
        ],
        "topic_attention": top_topics,
        "domain_attention": [
            {"name": item["domain"], "share": item["share"], "rel": item["rel"]}
            for item in top_topics
        ],
        "flags": flags,
        "proposal_values": proposal_values,
        "speech_types": [
            {"name": "Public remarks / address", "value": 70.0},
            {"name": "State of the Union / annual message", "value": 30.0},
        ],
        "adversaries": [{"name": "Named rival", "n": 7}],
        "ai_radar": ai_radar,
    }


def _make_profile_data(
    presidents: list[str],
    *,
    thin_presidents: set[str] | None = None,
) -> tuple[dict, list[str]]:
    thin_presidents = thin_presidents or set()
    display_issues = topic_quality.display_issues(BASE_ISSUES)
    score_rows = []
    issue_rows = []
    issue_cards = {}
    ai_by_president = {}
    distinctive_rows = []
    signatures = {}
    invokes = {}
    invoked_by = {}
    voice_neighbors = {}
    agenda_neighbors = {}
    invocation_v2 = {}
    feature_neighbors = {}

    for position, president in enumerate(presidents):
        thin = president in thin_presidents
        n_speeches = 1 if thin else 8 + position % 4
        row = {
            "party": PARTY.get(president, "Test party"),
            "first_year": 1789 + position * 4,
            "last_year": 1793 + position * 4,
            "n_speeches": n_speeches,
            "n_words": n_speeches * 1_250,
            "nrc_hope": 12.5 + position / 10,
            "nrc_fear": 7.5 + position / 10,
            "certainty": 0.62,
            "us_them": 8.2,
            "self_reference": 0.31,
            "fk_grade": 10.4,
            "ttr": 0.55,
            "religiosity": 4.6,
            "hype": 2.1,
            "mechanism": 8.0,
            "nonfinite_probe": np.inf,
        }
        for key, _ in profiles.RADAR_AXES:
            row[f"pct_{key}"] = 50.0 if thin else float(35 + position % 55)
        score_rows.append(row)

        issue_row = {"n_paragraphs": 120 + position}
        for issue_position, issue in enumerate(display_issues):
            issue_row[f"share_{issue}"] = round(0.16 - issue_position * 0.004, 4)
            issue_row[f"rel_{issue}"] = round(2.4 - issue_position * 0.1, 4)
        issue_rows.append(issue_row)

        cards = [
            {
                "issue": issue,
                "words": [f"term-{index + 1}"],
                "quote": f"Evidence excerpt {index + 1}.",
                "cite": f"Speech {index + 1} ({1900 + index})",
                "stance": "elevated attention",
                "share": 0.16 - index * 0.01,
                "rel": 2.4 - index * 0.2,
                "topic_of_day": index == 0,
            }
            for index, issue in enumerate(BASE_ISSUES)
        ]
        issue_cards[president] = {
            "n_paragraphs": 120 + position,
            "cards": cards,
            "voice": ["constitutional", "commonwealth"],
            "low_confidence": thin,
        }
        ai_by_president[president] = _ai_profile(position, n_speeches, thin)
        distinctive_rows.extend([
            {"president": president, "term": "stewardship", "z": 3.2, "rank": 0},
            {
                "president": president,
                "term": "union",
                "z": np.nan,
                "rank": 1,
            },
        ])
        speech_route = f"test-address-{profiles.slug(president)}"
        if position % 3 == 1:
            speech_route = profiles.MILLER_SPEECH_PATH + speech_route
        elif position % 3 == 2:
            speech_route = profiles.MILLER_ORIGIN + profiles.MILLER_SPEECH_PATH + speech_route
        signatures[president] = [{
            "title": "A signature address",
            "year": 1900 + position,
            "url": speech_route,
        }]
        invokes[president] = [{"target": "George Washington", "n": 2}]
        invoked_by[president] = {"total": 3}
        invocation_v2[president] = [{
            "target": "George Washington",
            "function": "legitimating precedent",
            "stance": "reverential",
            "mentions": 2,
        }]

    for position, president in enumerate(presidents):
        others = [candidate for candidate in presidents if candidate != president]
        neighbor = others[0] if others else president
        voice_neighbors[president] = [[neighbor, 0.91]] if others else []
        agenda_neighbors[president] = [[neighbor, 0.87]] if others else []
        feature_neighbors[president] = {
            key: ([{"president": neighbor, "similarity": 0.80 + position / 1000}]
                  if others else [])
            for key in profiles.FEATURE_SIMILARITY
        }

    data = {
        "scores": pd.DataFrame(score_rows, index=presidents),
        "issues": pd.DataFrame(issue_rows, index=presidents),
        "issue_meta": {"issues": BASE_ISSUES},
        "issue_cards": issue_cards,
        "ai": {"by_president": ai_by_president},
        "distinctive": pd.DataFrame(distinctive_rows),
        "signatures": signatures,
        "invokes": invokes,
        "invoked_by": invoked_by,
        "voice_neighbors": voice_neighbors,
        "agenda_neighbors": agenda_neighbors,
        "invocation_v2": invocation_v2,
        "feature_neighbors": feature_neighbors,
    }
    return data, display_issues


@pytest.fixture(scope="module")
def profile_bundle():
    data, display_issues = _make_profile_data(
        SMALL_PRESIDENTS,
        thin_presidents={"William Harrison"},
    )
    views = profiles_site.build_profile_view_models(data, display_issues)
    return data, display_issues, views


@pytest.fixture(scope="module")
def generated_profiles(tmp_path_factory):
    presidents = list(PARTY)
    data, display_issues = _make_profile_data(
        presidents,
        thin_presidents={"William Harrison", "Zachary Taylor", "James A. Garfield"},
    )
    views = profiles_site.build_profile_view_models(data, display_issues)
    site_dir = tmp_path_factory.mktemp("profile-site")
    written = profiles_site.write_profiles(data, site_dir, views=views)
    return site_dir, written


def _assert_finite_json(value) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite_json(nested)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_public_v3_contract_has_exact_keys_nested_shapes_and_finite_json(profile_bundle):
    _, _, views = profile_bundle
    view = views["Franklin D. Roosevelt"]
    payload = profiles_site.profile_public_payload(view)

    expected_keys = (
        "schema_version", "president", "slug", "party", "years", "sample",
        "rhetorical_radar", "raw_stats", "legacy_issue_attention",
        "issue_evidence", "ai", "distinctive_vocabulary", "signature_speeches",
        "legacy_invocations", "classified_invocations", "voice_neighbors",
        "agenda_neighbors", "feature_neighbors", "context_specific",
    )
    assert profiles_site.PROFILE_SCHEMA_VERSION == "president-profile-v3"
    assert profiles_site.PROFILE_PUBLIC_KEYS == expected_keys
    assert tuple(payload) == expected_keys
    assert set(view) == {*expected_keys, "_view"}
    assert set(payload["years"]) == {"first", "last"}
    assert set(payload["sample"]) == {
        "n_speeches", "n_words", "thin_record", "warning",
    }
    assert all(
        set(measure) == {"key", "label", "absolute", "percentile"}
        for measure in payload["rhetorical_radar"]
    )
    assert all(
        set(issue) == {"key", "label", "share", "era_relative_difference"}
        for issue in payload["legacy_issue_attention"]
    )
    assert set(payload["legacy_invocations"]) == {"invokes", "invoked_by"}
    assert set(payload["context_specific"]) == {"profile_only", "compare_only"}
    assert "_view" not in payload
    assert payload["raw_stats"]["nonfinite_probe"] is None
    assert payload["distinctive_vocabulary"][1]["z"] is None
    _assert_finite_json(payload)
    json.dumps(payload, allow_nan=False)


def test_measure_order_and_units_are_explicit_without_changing_v3(profile_bundle):
    _, _, views = profile_bundle
    view = views["Franklin D. Roosevelt"]
    legacy_rows = profiles_site.profile_measure_rows(view, "rhetorical_radar")
    ai_rows = profiles_site.profile_measure_rows(view, "ai_radar")

    assert [item["key"] for item in view["rhetorical_radar"]] == [
        key for key, _ in profiles.RADAR_AXES
    ]
    assert [row["key"] for row in legacy_rows] == [
        spec["key"] for spec in profiles_site.LEGACY_MEASURE_SPECS
    ]
    assert [row["key"] for row in ai_rows] == [
        spec["key"] for spec in profiles_site.AI_MEASURE_SPECS
    ]
    assert [row["unit"] for row in legacy_rows] == [
        spec["unit"] for spec in profiles_site.LEGACY_MEASURE_SPECS
    ]
    assert [row["unit"] for row in ai_rows] == [
        spec["unit"] for spec in profiles_site.AI_MEASURE_SPECS
    ]
    assert view["_view"]["units"] == {
        "rhetorical_radar": {
            spec["key"]: spec["unit"] for spec in profiles_site.LEGACY_MEASURE_SPECS
        },
        "ai_radar": {
            spec["key"]: spec["unit"] for spec in profiles_site.AI_MEASURE_SPECS
        },
        "legacy_issue_share": "fraction",
        "legacy_issue_era_relative_difference": "percentage points",
        "ai_topic_share": "percent",
        "ai_topic_era_relative_difference": "percentage points",
        "similarity": "cosine score",
    }


def test_president_percentiles_rank_only_five_speech_records():
    values = pd.Series({"lower": 10.0, "upper": 20.0, "thin_extreme": 10_000.0})
    speech_counts = pd.Series({"lower": 5, "upper": 6, "thin_extreme": 4})

    ranked = indices.eligible_percentile(values, speech_counts)

    assert indices.PRESIDENT_PERCENTILE_MIN_SPEECHES == 5
    assert profiles.SPARSE_MIN_SPEECHES == 5
    assert ranked["lower"] == pytest.approx(50.0)
    assert ranked["upper"] == pytest.approx(100.0)
    assert pd.isna(ranked["thin_extreme"])


def test_thin_profile_never_synthesizes_percentiles_or_figures(profile_bundle):
    _, _, views = profile_bundle
    view = views["William Harrison"]
    payload = profiles_site.profile_public_payload(view)

    assert payload["sample"] == {
        "n_speeches": 1,
        "n_words": 1_250,
        "thin_record": True,
        "warning": "Fewer than five corpus speeches: visible, but not ranked as a precise outlier.",
    }
    assert all(item["percentile"] is None for item in payload["rhetorical_radar"])
    assert all(
        payload["raw_stats"][f"pct_{key}"] is None for key, _ in profiles.RADAR_AXES
    )
    assert all(
        item["percentile"] is None
        for item in payload["ai"]["ai_radar"].values()
    )
    assert all(
        row["percentile"] is None
        and row["percentile_display"] == "Insufficient record"
        for layer in ("rhetorical_radar", "ai_radar")
        for row in profiles_site.profile_measure_rows(view, layer)
    )
    assert profiles_site.profile_figure_payload(view) == {}
    json.dumps(payload, allow_nan=False)


def test_six_topic_payload_copy_and_rendered_rows_agree(profile_bundle):
    _, _, views = profile_bundle
    view = views["Franklin D. Roosevelt"]
    page = profiles_site.render_profile(view)

    assert len(view["ai"]["top_topics"]) == 6
    assert view["_view"]["topic_count"] == 6
    assert page.count("data-topic-row") == 6
    assert "Six leading AI topics" in page
    assert "eight leading topics" not in page.lower()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
     (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
     (23, "23rd"), (83, "83rd"), (100, "100th")],
)
def test_shared_ordinal_formatter(value, expected):
    assert profiles.format_ordinal(value) == expected


def test_shared_percentile_and_speech_count_formatters():
    assert profiles.format_percentile(83) == "83rd percentile"
    assert profiles.format_percentile(None) == "Not ranked"
    assert profiles.format_percentile(np.nan, unavailable="Insufficient record") == (
        "Insufficient record"
    )
    assert profiles.format_speech_count(1) == "1 speech"
    assert profiles.format_speech_count(2) == "2 speeches"


def test_rich_and_thin_page_order_and_warning_placement(profile_bundle):
    _, _, views = profile_bundle
    rich = profiles_site.render_profile(views["Franklin D. Roosevelt"])
    thin = profiles_site.render_profile(views["William Harrison"])
    section_ids = [
        "overview", "agenda", "rhetoric", "connections", "similarity",
        "speeches", "evidence",
    ]

    assert [rich.index(f'<section id="{section_id}"') for section_id in section_ids] == sorted(
        rich.index(f'<section id="{section_id}"') for section_id in section_ids
    )
    assert "data-thin-record" not in rich
    assert thin.index('data-thin-record="true"') < thin.index('class="profile-nav"')
    assert thin.index('data-thin-record="true"') < thin.index('<section id="overview"')
    assert thin.index('<section id="overview"') < thin.index('<section id="rhetoric"')
    assert "Unsupported percentiles are shown as N/A and are not plotted." in thin
    assert "N/A · Insufficient record." in thin
    assert 'data-fig="' not in thin
    assert "<h2>Signature speeches</h2>" in rich
    assert "Chronology kept separate from synthesis" in rich


def test_navigation_evidence_similarity_and_speech_structure(profile_bundle):
    _, _, views = profile_bundle
    page = profiles_site.render_profile(views["Franklin D. Roosevelt"])

    for label, target in (
        ("Overview", "overview"), ("Agenda", "agenda"), ("Rhetoric", "rhetoric"),
        ("Connections", "connections"), ("Similarity", "similarity"),
        ("Speeches", "speeches"), ("Evidence", "evidence"),
    ):
        assert f'<a href="#{target}">{label}</a>' in page
    nav_links = [
        page.index(f'<a href="#{target}">{label}</a>')
        for label, target in (
            ("Overview", "overview"), ("Agenda", "agenda"), ("Rhetoric", "rhetoric"),
            ("Connections", "connections"), ("Similarity", "similarity"),
            ("Speeches", "speeches"), ("Evidence", "evidence"),
        )
    ]
    assert nav_links == sorted(nav_links)
    assert ".profile-page :where(a, button, summary) { min-width: 44px; min-height: 44px; }" in page
    speeches = page.index('<section id="speeches"')
    evidence = page.index('<section id="evidence"')
    download = page.index('<p class="download-row">')
    adjacent_navigation = page.index('aria-label="Adjacent president profiles"')
    assert speeches < evidence < download < adjacent_navigation
    evidence_disclosure = page.index('<details class="evidence-section-disclosure">')
    assert evidence < page.index("<h2>Evidence</h2>", evidence) < evidence_disclosure
    assert "<details class=\"evidence-section-disclosure\" open" not in page
    assert evidence_disclosure < page.index("Legacy issue model · named artifacts")
    first = page.index("Evidence excerpt 1.")
    second = page.index("Evidence excerpt 2.")
    disclosure = page.index("More issue evidence (1)")
    third = page.index("Evidence excerpt 3.")
    assert first < second < disclosure < third
    assert page.count("data-similarity=") == 5
    assert page.count('class="similarity-kicker"') == 5
    assert 'class="similarity-match"' in page
    assert ".similarity-match { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 10px; align-items: center; min-height: 44px; }" in page
    assert ".similarity-card li a { display: flex; min-height: 44px; align-items: center;" in page
    assert ".similarity-card data { display: inline-flex; min-height: 30px; align-items: center; justify-content: center;" in page
    assert ".similarity-card data { float:" not in page
    assert "not combined into an overall likeness" in page
    assert '<ol class="signature-list">' in page
    assert 'aria-label="Adjacent president profiles"' in page
    assert "Corpus-based presidential profile" in page
    assert "Evidence-first presidential profile" not in page


def test_semantic_rhetoric_bars_and_closed_exact_tables_replace_plotly(profile_bundle):
    _, _, views = profile_bundle
    view = views["Franklin D. Roosevelt"]
    figures = profiles_site.profile_figure_payload(view)
    page = profiles_site.render_profile(view)
    assert figures == {}
    assert 'data-fig="' not in page
    for key in ("rhetoric_legacy", "rhetoric_ai"):
        assert f'id="{key}-title"' in page
        assert f'class="chart-summary" id="{key}-summary"' in page
    assert page.count('<ol class="percentile-bars"') == 2
    assert page.count("Exact values and eligible-president ranks</summary>") == 2
    assert page.count("exact values and eligible-president ranks</caption>") == 2
    assert page.count('scope="col"') >= 6
    assert page.count('scope="row"') == (
        len(profiles_site.LEGACY_MEASURE_SPECS) + len(profiles_site.AI_MEASURE_SPECS)
    )
    assert "Plotly" not in page
    assert "plotly" not in page.lower()
    assert "cdn.plot.ly" not in page


def test_mobile_and_keyboard_accessibility_guards_are_in_profile_html(profile_bundle):
    _, _, views = profile_bundle
    page = profiles_site.render_profile(views["Franklin D. Roosevelt"])

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert "body.profile-page { overflow-x: clip; }" in page
    assert "@media (max-width: 760px)" in page
    assert "@media (max-width: 600px)" in page
    assert '<span class="profile-nav-cue" aria-hidden="true">More sections →</span>' in page
    assert ".profile-nav-cue { display: inline-flex; min-height: 24px" in page
    assert ".profile-page :where(.eyebrow, .glance-card span, .source-tag" in page
    assert ".president-nav span, .invocation-cards dt) { font-size: 12px!important; }" in page
    assert ".profile-nav ul { display: flex; width: max-content" in page
    assert ".profile-nav a { min-width: 92px; min-height: 44px" in page
    assert ".measure-table td::before { display: block; color: var(--muted); font-size: 12px" in page
    assert ".percentile-row" in page
    assert ".measure-table, .measure-table tbody { display: block; }" in page
    assert "overflow-wrap: anywhere" in page
    assert ":focus-visible" in page
    assert "outline: 3px solid var(--focus)" in page
    assert 'aria-label="On this profile"' in page


def test_directory_is_chronological_name_only_and_progressive(profile_bundle):
    data, display_issues, _ = profile_bundle
    page = profiles_site.render_index(data, display_issues)

    assert '<ol class="directory-grid">' in page
    assert page.count("<li data-president-card") == len(SMALL_PRESIDENTS)
    assert [page.index(profiles.slug(name) + ".html") for name in SMALL_PRESIDENTS] == sorted(
        page.index(profiles.slug(name) + ".html") for name in SMALL_PRESIDENTS
    )
    assert "Legacy issue" not in page
    assert "Top AI topic by source-document paragraph share" in page
    assert "Corpus record " in page
    assert 'data-directory-search hidden' in page
    assert "card.dataset.searchName.includes(query)" in page
    assert "card.hidden = !matches" in page
    assert 'form.addEventListener("submit"' in page
    assert "event.preventDefault(); apply();" in page
    assert "toLocaleLowerCase" in page
    assert 'event.key !== "Escape"' in page
    assert "history." not in page
    assert "localStorage" not in page
    assert "grid-template-columns: repeat(3" in page
    assert ".card-measure span, .card-support { font-size: 12px; }" in page
    assert "@media (max-width: 959px)" in page
    assert "@media (max-width: 599px)" in page


def test_connections_module_fallback_keeps_the_same_600px_lazy_boundary():
    source = inspect.getsource(profiles_site._connections_content)
    assert "bounds.top > innerHeight + 600" in source
    assert 'addEventListener("scroll", maybeLoad' in source
    assert 'addEventListener("resize", maybeLoad' in source
    assert "else {\n    load().catch" not in source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("inaugural-address", "https://millercenter.org/the-presidency/presidential-speeches/inaugural-address"),
        ("the-presidency/presidential-speeches/inaugural-address", "https://millercenter.org/the-presidency/presidential-speeches/inaugural-address"),
        ("/the-presidency/presidential-speeches/inaugural-address", "https://millercenter.org/the-presidency/presidential-speeches/inaugural-address"),
        ("/the-presidency/presidential-speeches/the-presidency/presidential-speeches/inaugural-address", "https://millercenter.org/the-presidency/presidential-speeches/inaugural-address"),
        ("https://millercenter.org/the-presidency/presidential-speeches/inaugural-address?x=1#text", "https://millercenter.org/the-presidency/presidential-speeches/inaugural-address?x=1#text"),
        ("https://example.test/already-absolute", "https://example.test/already-absolute"),
    ],
)
def test_miller_signature_urls_are_normalized_once(source, expected):
    assert profiles.miller_speech_url(source) == expected


@pytest.mark.parametrize(
    "source",
    [None, "", "javascript:alert(1)", "https://example.test/not-corpus"],
)
def test_profile_source_urls_fail_closed_outside_miller_speech_corpus(source):
    with pytest.raises(ValueError, match="Profile source URL"):
        profiles_site._normalized_profile_source_url(source)


def test_enriched_evidence_renders_exact_denominator_and_method_copy():
    info = {
        "method_definitions": {
            "legacy_stance_v3": "Declared stance method copy.",
            "legacy_distinctive_vocabulary_v3": "Declared vocabulary method copy.",
        },
        "cards": [{
            "issue": "Economic policy",
            "legacy_v3": {
                "issue": "Economic policy", "share": .2, "rel": 1.0,
                "base": .1, "topic_of_day": False, "words": ["work"],
                "quote": None, "cite": None, "stance": "supportive",
            },
            "claim": {"type": "absolute_and_era_relative_emphasis", "text": "Both thresholds passed."},
            "exact_evidence": {
                "issue_paragraph_count": 20,
                "total_document_owned_paragraph_count": 100,
                "percentage": 20.0,
                "source_document_count": 3,
                "corpus_baseline_percentage": 10.0,
                "corpus_baseline_multiple": 2.0,
                "era_difference_percentage_points": 1.0,
            },
            "why_shown": {"text": "The declared absolute and era thresholds passed."},
            "receipts": [],
            "method": {
                "stance": "supportive",
                "stance_method": "legacy_stance_v3",
                "distinctive_vocabulary": ["work"],
                "distinctive_vocabulary_method": "legacy_distinctive_vocabulary_v3",
            },
            "limitation": "Document-owner limitation.",
        }],
    }

    rendered = profiles_site._evidence_cards_from_info(info)

    assert "20 / 100 (20.0%)" in rendered
    assert "Declared stance method copy." in rendered
    assert "Declared vocabulary method copy." in rendered
    assert "legacy_stance_v3</p>" not in rendered


def test_enriched_evidence_receipts_are_keyed_attributed_disclosed_and_escaped():
    receipt = {
        "doc_name": "test-address",
        "para_idx": 7,
        "title": "Test address",
        "speech_date": "1900-01-02",
        "year": 1900,
        "source_url": (
            "https://millercenter.org/the-presidency/presidential-speeches/"
            "test-address"
        ),
        "source_document_owner": "Owner <unsafe>",
        "source_document_owner_profile_id": "owner-id",
        "actual_speaker": "Speaker <unsafe>",
        "actual_speaker_profile_id": "speaker-id",
        "cross_owner": True,
        "speaker_eligibility_state": "eligible",
        "speaker_exclusion_reason": None,
        "excerpt": "</blockquote><script>alert(1)</script>",
        "selection_role": "primary",
    }
    card = {
        "issue": "Economic policy",
        "legacy_v3": {
            "issue": "Economic policy", "share": .2, "rel": 1.0,
            "base": .1, "topic_of_day": False, "words": [],
            "quote": None, "cite": None, "stance": None,
        },
        "claim": {"text": "Both thresholds passed."},
        "exact_evidence": {
            "issue_paragraph_count": 20,
            "total_document_owned_paragraph_count": 100,
            "percentage": 20.0,
            "source_document_count": 3,
            "corpus_baseline_percentage": 10.0,
            "corpus_baseline_multiple": 2.0,
            "era_difference_percentage_points": 1.0,
        },
        "why_shown": {"text": "Both declared thresholds passed."},
        "receipts": [
            receipt,
            {**receipt, "doc_name": "second-address", "para_idx": 2,
             "source_url": "https://millercenter.org/the-presidency/presidential-speeches/second-address",
             "selection_role": "additional_1"},
            {**receipt, "doc_name": "third-address", "para_idx": 3,
             "source_url": "https://millercenter.org/the-presidency/presidential-speeches/third-address",
             "selection_role": "additional_2"},
        ],
        "method": {},
        "limitation": "Document-owner limitation.",
    }

    rendered = profiles_site._evidence_cards_from_info({
        "cards": [card],
        "thin_record_warning": "Fewer than five source speeches are present.",
    })

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "(test-address, 7)" in rendered
    assert "Corpus year 1900" in rendered
    assert "Owner &lt;unsafe&gt; (owner-id)" in rendered
    assert "Speaker &lt;unsafe&gt; (speaker-id)" in rendered
    assert "Cross-owner: actual speaker differs" in rendered
    assert "Additional keyed receipts (2)" in rendered
    assert rendered.index("test-address") < rendered.index("second-address")
    assert "Thin source-document record." in rendered
    assert "Document-owner limitation." in rendered


def test_all_45_unique_profiles_and_public_json_are_written(generated_profiles):
    site_dir, views = generated_profiles
    html_files = sorted((site_dir / "presidents").glob("*.html"))
    json_files = sorted((site_dir / "data" / "presidents").glob("*.json"))

    assert len(views) == len(PARTY) == 45
    assert len({view["slug"] for view in views.values()}) == 45
    assert len([path for path in html_files if path.name != "index.html"]) == 45
    assert len(json_files) == 45
    assert {path.stem for path in json_files} == {view["slug"] for view in views.values()}
    directory = (site_dir / "presidents" / "index.html").read_text(encoding="utf-8")
    assert directory.count("<li data-president-card") == 45
    assert directory.count('loading="eager"') == 6
    assert directory.count('loading="lazy"') == 39
    assert directory.count('decoding="async"') == 45
    assert len(directory.encode("utf-8")) <= 45_000
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert tuple(payload) == profiles_site.PROFILE_PUBLIC_KEYS
        assert payload["schema_version"] == "president-profile-v3"
        _assert_finite_json(payload)


def test_real_president_profile_v3_payloads_remain_value_for_value_unchanged(monkeypatch):
    annotation_dir = REPO_ROOT / "data" / "llm_annotations"
    monkeypatch.setattr(llm_annotations, "ANNOTATIONS_DIR", annotation_dir)
    monkeypatch.setattr(ai_labels, "AGREEMENT_PATH", annotation_dir / "agreement_v1.parquet")
    data = profiles.build_profile_data()
    data["ai"] = ai_labels.build_ai_data()
    data["feature_neighbors"] = profiles.feature_neighbors(data, data["ai"])
    display_issues = topic_quality.display_issues(data["issue_meta"]["issues"])
    views = profiles_site.build_profile_view_models(
        data, display_issues, require_complete=True,
    )
    directory = profiles_site.render_index(data, display_issues)

    for view in views.values():
        expected_path = REPO_ROOT / "docs" / "data" / "presidents" / f"{view['slug']}.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        assert profiles_site.profile_public_payload(view) == expected
        card_start = directory.index(f'href="{view["slug"]}.html"')
        card = directory[card_start:directory.index("</li>", card_start)]
        leading = view["_view"]["leading_ai_topic"]
        assert html_lib.escape(str(leading["name"])) in card
        assert f'{float(leading["share"]):.1f}%' in card


def test_generated_portrait_compare_json_signature_and_profile_links_are_valid(
    generated_profiles,
):
    site_dir, views = generated_profiles
    president_dir = site_dir / "presidents"
    data_dir = site_dir / "data" / "presidents"

    for president, view in views.items():
        page_path = president_dir / f'{view["slug"]}.html'
        page = page_path.read_text(encoding="utf-8")

        portrait = re.search(r'<img class="portrait" src="([^"]+)"', page).group(1)
        assert portrait == f'../portraits/{view["slug"]}.png'
        assert (REPO_ROOT / "docs" / "portraits" / f'{view["slug"]}.png').is_file()

        compare_href = re.search(r'class="compare-action" href="([^"]+)"', page).group(1)
        compare_url = urlsplit(compare_href.replace("&amp;", "&"))
        assert compare_url.path == "../compare.html"
        assert parse_qs(compare_url.query)["a"] == [view["slug"]]

        json_href = re.search(
            r'href="(../data/presidents/[^"]+\.json)" download', page
        ).group(1)
        assert json_href == f'../data/presidents/{view["slug"]}.json'
        assert (data_dir / f'{view["slug"]}.json').is_file()

        signature_href = re.search(
            r'<ol class="signature-list">.*?<a href="([^"]+)"', page, re.DOTALL
        ).group(1)
        assert urlsplit(signature_href).scheme == "https"
        assert urlsplit(signature_href).netloc == "millercenter.org"
        assert signature_href.count(profiles.MILLER_SPEECH_PATH) == 1

        for adjacent in re.findall(
            r'<a class="(?:previous|next)" href="([^"]+\.html)"', page
        ):
            assert (president_dir / adjacent).is_file()


def test_final_generated_profile_budget_validator_covers_post_shell_inventory(tmp_path):
    president_dir = tmp_path / "presidents"
    shard_dir = tmp_path / "data" / "profile-context" / "presidents"
    asset_dir = tmp_path / "assets"
    president_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    asset_dir.mkdir(parents=True)
    (president_dir / "index.html").write_text("<html><body>Directory</body></html>")
    for index in range(45):
        slug_value = f"president-{index}"
        (president_dir / f"{slug_value}.html").write_text(
            '<html><body><img src="../portraits/example.png"></body></html>'
        )
        (shard_dir / f"{slug_value}_v1.json").write_text("{}\n")
    (tmp_path / "data" / "profile-context" / "index_v1.json").write_text("{}\n")
    (asset_dir / "profile-connections-v1.js").write_text("export const ready = true;\n")

    report = profiles_site.validate_generated_profile_budgets(tmp_path)

    assert report["profiles"] == 45
    assert report["maximum_profile_raw_bytes"] > 0
