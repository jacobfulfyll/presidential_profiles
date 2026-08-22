"""Contract and rendering regressions for evidence-first president profiles."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest

from presidential_profiles import indices, profiles, profiles_site, topic_quality
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
    section_ids = ["overview", "agenda", "rhetoric", "evidence", "similarity", "speeches"]

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
        ("Overview", "overview"), ("Agenda", "agenda"), ("Evidence", "evidence"),
        ("Similarity", "similarity"), ("Speeches", "speeches"),
    ):
        assert f'<a href="#{target}">{label}</a>' in page
    first = page.index("Evidence excerpt 1.")
    second = page.index("Evidence excerpt 2.")
    disclosure = page.index("Additional issue evidence (1)")
    third = page.index("Evidence excerpt 3.")
    assert first < second < disclosure < third
    assert page.count("data-similarity=") == 5
    assert "not combined into an overall likeness" in page
    assert '<ol class="signature-list">' in page
    assert 'aria-label="Adjacent president profiles"' in page


def test_chart_names_summaries_tables_and_figure_keys_match(profile_bundle):
    _, _, views = profile_bundle
    view = views["Franklin D. Roosevelt"]
    figures = profiles_site.profile_figure_payload(view)
    page = profiles_site.render_profile(view)
    rendered_keys = re.findall(r'data-fig="([^"]+)"', page)

    assert list(figures) == ["rhetoric_legacy", "rhetoric_ai"]
    assert rendered_keys == list(figures)
    for key in figures:
        assert (
            f'data-fig="{key}" role="img" aria-labelledby="{key}-title" '
            f'aria-describedby="{key}-summary"'
        ) in page
        assert f'id="{key}-title"' in page
        assert f'class="chart-summary" id="{key}-summary"' in page
    assert page.count("structured data alternative</caption>") == 2
    assert page.count('scope="col"') >= 6
    assert page.count('scope="row"') == (
        len(profiles_site.LEGACY_MEASURE_SPECS) + len(profiles_site.AI_MEASURE_SPECS)
    )
    assert '<script src="../assets/plotly-3.0.1.min.js"></script>' in page
    assert "cdn.plot.ly" not in page


def test_mobile_and_keyboard_accessibility_guards_are_in_profile_html(profile_bundle):
    _, _, views = profile_bundle
    page = profiles_site.render_profile(views["Franklin D. Roosevelt"])

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert "body.profile-page { overflow-x: clip; }" in page
    assert "@media (max-width: 600px)" in page
    assert ".profile-figure { display: none; }" in page
    assert ".measure-table, .measure-table tbody { display: block; }" in page
    assert "overflow-wrap: anywhere" in page
    assert ":focus-visible" in page
    assert "outline: 3px solid var(--focus)" in page
    assert 'aria-label="On this profile"' in page


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


def test_all_45_unique_profiles_and_public_json_are_written(generated_profiles):
    site_dir, views = generated_profiles
    html_files = sorted((site_dir / "presidents").glob("*.html"))
    json_files = sorted((site_dir / "data" / "presidents").glob("*.json"))

    assert len(views) == len(PARTY) == 45
    assert len({view["slug"] for view in views.values()}) == 45
    assert len([path for path in html_files if path.name != "index.html"]) == 45
    assert len(json_files) == 45
    assert {path.stem for path in json_files} == {view["slug"] for view in views.values()}
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert tuple(payload) == profiles_site.PROFILE_PUBLIC_KEYS
        assert payload["schema_version"] == "president-profile-v3"
        _assert_finite_json(payload)


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
