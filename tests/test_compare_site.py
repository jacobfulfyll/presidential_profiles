"""Contract tests for the native, progressively enhanced Compare V2 page."""

from __future__ import annotations

from copy import deepcopy
import html
import json
from pathlib import Path
import re

import pytest

from presidential_profiles import compare_site, profiles, profiles_site


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "docs" / "data" / "presidents"
HOSTILE = '</script><img src=x onerror="alert(1)">'


@pytest.fixture(scope="module")
def public_profiles() -> dict[str, dict]:
    return {
        president: json.loads(
            (PROFILE_DIR / f"{profiles.slug(president)}.json").read_text(
                encoding="utf-8"
            )
        )
        for president in profiles.PARTY
    }


@pytest.fixture(scope="module")
def compare_payload(public_profiles) -> dict:
    return compare_site.comparison_payload(
        public_profiles,
        require_complete=True,
    )


@pytest.fixture(scope="module")
def compare_page(compare_payload) -> str:
    return compare_site.render_compare_page(compare_payload)


def test_exact_population_order_slugs_and_public_display_names(compare_payload):
    order = list(profiles.PARTY)
    projected = compare_payload["presidents"]

    assert compare_payload["schema_version"] == "president-comparison-page-v2"
    assert compare_payload["profile_schema_version"] == "president-profile-v3"
    assert compare_payload["order"] == order
    assert list(projected) == order
    assert len(projected) == 45
    assert len({item["slug"] for item in projected.values()}) == 45
    assert [projected[name]["slug"] for name in order] == [
        profiles.slug(name) for name in order
    ]
    assert projected["William Harrison"]["display_name"] == "William Henry Harrison"
    assert projected["William Taft"]["display_name"] == "William Howard Taft"
    assert all(
        item["president"] == president
        for president, item in projected.items()
    )


def test_payload_is_compact_finite_and_omits_public_profile_bulk(compare_payload):
    encoded = json.dumps(
        compare_payload,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert len(encoded) < 300_000
    assert set(compare_payload["presidents"]["Abraham Lincoln"]) == {
        "president", "display_name", "slug", "party", "years", "sample",
        "measures", "agenda", "neighbors", "evidence",
    }
    assert "raw_stats" not in encoded.decode("utf-8")
    assert "context_specific" not in encoded.decode("utf-8")
    assert "issue_evidence" not in encoded.decode("utf-8")


def test_measure_catalogs_and_values_are_exact_profile_v3_rows(
    public_profiles, compare_payload
):
    expected_catalogs = {
        "corpus": profiles_site.LEGACY_MEASURE_SPECS,
        "ai": profiles_site.AI_MEASURE_SPECS,
    }
    for layer, specs in expected_catalogs.items():
        assert [row["key"] for row in compare_payload["catalogs"]["measures"][layer]] == [
            spec["key"] for spec in specs
        ]
        assert [row["label"] for row in compare_payload["catalogs"]["measures"][layer]] == [
            spec["label"] for spec in specs
        ]
        assert [row["unit"] for row in compare_payload["catalogs"]["measures"][layer]] == [
            spec["unit"] for spec in specs
        ]

    for president in ("Abraham Lincoln", "Franklin D. Roosevelt", "William Harrison"):
        source = public_profiles[president]
        projected = compare_payload["presidents"][president]
        for output_layer, profile_layer in (
            ("corpus", "rhetorical_radar"),
            ("ai", "ai_radar"),
        ):
            rows = profiles_site.profile_measure_rows(source, profile_layer)
            for expected, cell in zip(rows, projected["measures"][output_layer], strict=True):
                assert cell[0] == expected["absolute"]
                assert cell[1] == expected["absolute_display"]
                assert cell[2] == expected["percentile"]
                expected_rank = (
                    "N/A · Insufficient record"
                    if source["sample"]["thin_record"]
                    else expected["percentile_display"]
                )
                assert cell[3] == expected_rank

    thin = compare_payload["presidents"]["William Harrison"]
    assert all(
        cell[0] is not None
        and cell[2] is None
        and cell[3] == "N/A · Insufficient record"
        for layer in ("corpus", "ai")
        for cell in thin["measures"][layer]
    )


def test_missing_ai_supported_null_rank_and_recorded_zero_are_distinct(
    public_profiles, compare_payload
):
    missing_ai = deepcopy(public_profiles["Abraham Lincoln"])
    missing_ai["ai"] = None
    missing_payload = compare_site.comparison_payload(
        {"Abraham Lincoln": missing_ai}
    )
    ai_cells = missing_payload["presidents"]["Abraham Lincoln"]["measures"]["ai"]
    assert all(cell == [None, "N/A", None, "N/A · AI labels unavailable"] for cell in ai_cells)
    assert missing_payload["presidents"]["Abraham Lincoln"]["agenda"]["topics"] is None

    thin_missing_ai = deepcopy(public_profiles["William Harrison"])
    thin_missing_ai["ai"] = None
    thin_missing_payload = compare_site.comparison_payload(
        {"William Harrison": thin_missing_ai}
    )
    assert all(
        cell == [None, "N/A", None, "N/A · AI labels unavailable"]
        for cell in thin_missing_payload["presidents"]["William Harrison"]["measures"]["ai"]
    )

    null_rank = deepcopy(public_profiles["Abraham Lincoln"])
    null_rank["rhetorical_radar"][0]["percentile"] = None
    null_payload = compare_site.comparison_payload({"Abraham Lincoln": null_rank})
    assert null_payload["presidents"]["Abraham Lincoln"]["measures"]["corpus"][0][3] == "N/A · Not ranked"

    # A real canonical zero remains a numeric zero rather than an unavailable cell.
    assert any(
        pair[0] == 0
        for profile in compare_payload["presidents"].values()
        for layer in ("domains", "topics", "legacy")
        for pair in (profile["agenda"][layer] or [])
    )


def _synthetic_agenda_payload() -> dict:
    catalogs = {
        "domains": [
            {"name": "Alpha"}, {"name": "Beta"},
            {"name": "Gamma"}, {"name": "Delta"},
        ],
        "topics": [],
        "legacy": [
            {"name": "Issue A", "key": "a"},
            {"name": "Issue B", "key": "b"},
            {"name": "Issue C", "key": "c"},
        ],
    }
    return {
        "catalogs": {"agenda": catalogs},
        "presidents": {
            "A": {"display_name": "President A", "agenda": {
                "domains": [[5, 1], [5, 2], [0, 0], [1, -1]],
                "topics": [],
                "legacy": [[10, -1], [0, 4], [2, 1]],
            }},
            "B": {"display_name": "President B", "agenda": {
                "domains": [[0, 0], [0, 0], [5, 2], [5, 3]],
                "topics": [],
                "legacy": [[1, 3], [8, -2], [0, 0]],
            }},
        },
    }


def test_shared_agenda_unions_caps_ties_positive_filter_and_zeros():
    payload = _synthetic_agenda_payload()
    domains = compare_site.shared_agenda_rows(
        payload, ["A", "B"], "domains", top_n=2, cap=3
    )
    # Every maximum is five, so canonical taxonomy order breaks the tie.
    assert [row["name"] for row in domains] == ["Alpha", "Beta", "Gamma"]

    all_domains = compare_site.shared_agenda_rows(
        payload, ["A", "B"], "domains", top_n=4, cap=None
    )
    gamma = next(row for row in all_domains if row["name"] == "Gamma")
    assert gamma["values"]["A"]["share"] == 0
    assert gamma["values"]["A"] is not None

    nullable = deepcopy(payload)
    nullable["presidents"]["A"]["agenda"]["domains"] = [
        [None, None], [0, 0], [None, None], [None, None]
    ]
    nullable["presidents"]["B"]["agenda"]["domains"] = [
        [None, None], [None, None], [None, None], [None, None]
    ]
    nullable_rows = compare_site.shared_agenda_rows(
        nullable, ["A", "B"], "domains", top_n=1, cap=None
    )
    assert [row["name"] for row in nullable_rows] == ["Beta", "Alpha"]
    nullable_table = compare_site._agenda_table(
        nullable, ["A", "B"], nullable_rows, "Nullable fixture",
        relative_label="vs era",
    )
    assert "N/A%" not in nullable_table
    assert "Not recorded" in nullable_table

    legacy = compare_site.shared_agenda_rows(
        payload,
        ["A", "B"],
        "legacy",
        top_n=4,
        cap=12,
        positive_relative=True,
        ranking_key="rel",
    )
    assert [row["name"] for row in legacy] == ["Issue B", "Issue A", "Issue C"]
    assert all(
        any((row["values"][name] or {}).get("rel", 0) > 0 for name in ("A", "B"))
        for row in legacy
    )


def test_production_agenda_contracts_use_full_canonical_arrays(compare_payload):
    selected = ["Abraham Lincoln", "Franklin D. Roosevelt", "Barack Obama"]
    domains = compare_site.shared_agenda_rows(
        compare_payload, selected, "domains", top_n=4, cap=12
    )
    topics = compare_site.shared_agenda_rows(
        compare_payload, selected, "topics", top_n=3, cap=9
    )
    extended_topics = compare_site.shared_agenda_rows(
        compare_payload, selected, "topics", top_n=6, cap=None
    )
    legacy = compare_site.shared_agenda_rows(
        compare_payload, selected, "legacy", top_n=4, cap=12,
        positive_relative=True, ranking_key="rel",
    )

    assert len(compare_payload["catalogs"]["agenda"]["domains"]) == 17
    assert len(compare_payload["catalogs"]["agenda"]["topics"]) == 50
    assert len(compare_payload["catalogs"]["agenda"]["legacy"]) == 16
    assert 1 <= len(domains) <= 12
    assert 1 <= len(topics) <= 9
    assert set(row["name"] for row in topics) <= set(
        row["name"] for row in extended_topics
    )
    assert len(legacy) <= 12


def test_server_rendered_default_structure_accessibility_and_size(compare_page):
    h2s = re.findall(r"<h2>(.*?)</h2>", compare_page)
    assert h2s == [
        "Overview", "Rhetoric", "Agenda",
        "Nearest-neighbor context", "Evidence &amp; data",
    ]
    assert "<fieldset" in compare_page and "<legend>" in compare_page
    for label in ("President A", "President B", "Optional President C"):
        assert f">{label}</label>" in compare_page
    assert '<option value="" selected>Not selected</option>' in compare_page
    for control in ("Swap A/B", "Add third president", "Copy link", "Reset"):
        assert f">{control}</button>" in compare_page
    assert 'role="status" aria-live="polite"' in compare_page
    assert "How this comparison is measured" in compare_page
    assert "Corpus record span" in compare_page
    assert "Abraham Lincoln" in compare_page
    assert "Franklin D. Roosevelt" in compare_page
    assert "All eight corpus-derived Profile V3 measures" in compare_page
    assert "All six AI-labeled Profile V3 measures" in compare_page
    assert compare_page.count("<caption>") >= 8
    assert compare_page.count('scope="col"') >= 20
    assert compare_page.count('scope="row"') >= 30
    assert "slot-marker slot-a" in compare_page
    assert "slot-marker slot-b" in compare_page
    assert "profile evidence follows document ownership" in compare_page
    assert "speaker-attribution audit is pending" not in compare_page
    assert "completed speaker-attribution audit" in compare_page
    assert "AI-labeled · exploratory" in compare_page
    assert "Legacy invocation counts" in compare_page
    assert "Stance:" in compare_page
    assert re.search(r">\+[0-9.]+ pp vs (?:era|contemporaries)<", compare_page)
    assert "class=\"win\"" not in compare_page
    assert '<script src="assets/plotly-3.0.1.min.js"></script>' in compare_page
    assert 'id="corpus-radar"' in compare_page
    assert 'id="ai-radar"' in compare_page
    assert 'role="tablist" aria-label="Rhetoric radar view"' in compare_page
    assert 'id="radar-tab-corpus" type="button" role="tab" aria-selected="true"' in compare_page
    assert 'id="radar-tab-ai" type="button" role="tab" aria-selected="false"' in compare_page
    assert 'id="radar-panel-corpus" role="tabpanel" aria-labelledby="radar-tab-corpus"' in compare_page
    assert 'id="radar-panel-ai" role="tabpanel" aria-labelledby="radar-tab-ai" hidden' in compare_page
    assert 'if (event.key==="ArrowRight")' in compare_page
    assert 'if (event.key==="ArrowLeft")' in compare_page
    assert 'if (event.key==="Home")' in compare_page
    assert 'if (event.key==="End")' in compare_page
    assert 'panel.hidden=!selected' in compare_page
    assert 'type:"scatterpolar"' in compare_page
    assert "Exact measures, units, and percentile states" in compare_page
    assert "@media (max-width:760px)" in compare_page
    assert "content:attr(data-label)" in compare_page
    assert "top:var(--global-nav-height,0px)" in compare_page
    assert "scroll-margin-top:calc(var(--global-nav-height,0px) + 70px)" in compare_page
    assert len(compare_page.encode("utf-8")) < 500_000


def test_five_similarity_instruments_remain_separate_and_linked(compare_page):
    for spec in profiles.FEATURE_SIMILARITY.values():
        assert html.escape(spec["label"]) in compare_page
        assert html.escape(spec["description"]) in compare_page
    assert "No composite score is synthesized" in compare_page
    assert compare_page.count('class="similarity-panel"') >= 5
    assert 'class="neighbor-meter"' in compare_page
    assert re.search(r'portraits/[a-z-]+\.png', compare_page)
    assert re.search(r'href="presidents/[a-z-]+\.html"', compare_page)


def test_evidence_dashboard_is_visual_compact_and_exact(compare_page):
    assert 'class="footprint-visual"' in compare_page
    assert "Corpus footprint" in compare_page
    assert compare_page.count('class="footprint-group"') >= 3
    assert 'class="signal-meter"' in compare_page
    assert 'class="term-list"' in compare_page
    assert 'class="speech-tile"' in compare_page
    assert "Read source excerpts (" in compare_page
    assert "Full evidence" in compare_page
    assert "Canonical JSON" in compare_page


def test_url_normalization_default_partial_last_invalid_duplicate_and_third():
    order = list(profiles.PARTY)
    by_slug = {profiles.slug(name): name for name in order}

    assert compare_site.normalize_selection(order, by_slug, {}) == (
        ["Abraham Lincoln", "Franklin D. Roosevelt"], False
    )
    assert compare_site.normalize_selection(
        order, by_slug, {"a": "abraham-lincoln"}
    ) == (["Abraham Lincoln", "Andrew Johnson"], False)
    assert compare_site.normalize_selection(
        order, by_slug, {"a": profiles.slug(order[-1])}
    ) == ([order[-1], order[-2]], False)

    normalized, corrected = compare_site.normalize_selection(
        order, by_slug, {"a": "not-a-president", "b": "also-invalid"}
    )
    assert normalized == ["Abraham Lincoln", "Franklin D. Roosevelt"]
    assert corrected is True

    assert compare_site.normalize_selection(
        order, by_slug, {"a": "not-a-president"}
    ) == (["Abraham Lincoln", "Franklin D. Roosevelt"], True)

    normalized, corrected = compare_site.normalize_selection(
        order,
        by_slug,
        {"a": "abraham-lincoln", "b": "abraham-lincoln", "c": "franklin-d-roosevelt"},
    )
    assert len(normalized) == 3
    assert len(set(normalized)) == 3
    assert corrected is True


def test_history_controls_hash_retention_and_download_contract(compare_page):
    assert "history.replaceState" in compare_page
    assert "history.pushState" in compare_page
    assert 'window.addEventListener("popstate"' in compare_page
    assert 'document.getElementById("swap-presidents")' in compare_page
    assert 'document.getElementById("toggle-third")' in compare_page
    assert "const hash = url.hash" in compare_page
    assert "url.hash = hash" in compare_page
    assert "option.disabled = selected.some" in compare_page
    assert 'empty.textContent = "Not selected"' in compare_page
    assert 'hint.textContent=index===2 && !selected[2]' in compare_page
    assert "const rankValue = value => Number.isFinite(value) ? value : -Infinity" in compare_page
    assert "formatSignedNumber(value.rel)" in compare_page

    assert 'const sourcePath=`data/presidents/${P[name].slug}.json`' in compare_page
    assert "await fetch(sourcePath)" in compare_page
    assert 'profile.schema_version!=="president-profile-v3"' in compare_page
    assert "profile.slug!==P[name].slug" in compare_page
    assert "source_url:new URL(sourcePath,location.href).href" in compare_page
    assert 'schema_version:DOWNLOAD_SCHEMA_VERSION' in compare_page
    assert 'selection_order:presidents.map(item=>item.slug)' in compare_page
    assert 'source_urls:presidents.map(item=>item.source_url)' in compare_page
    assert "president-comparison-v2-${presidents.map" in compare_page
    assert "Preparing comparison download…" in compare_page
    assert "Comparison download ready." in compare_page
    assert "Comparison download failed." in compare_page


def test_all_projected_visible_strings_are_escaped_and_script_safe(public_profiles):
    hostile = deepcopy(public_profiles["Abraham Lincoln"])
    hostile["party"] = HOSTILE
    hostile["legacy_issue_attention"][0]["label"] = HOSTILE
    hostile["ai"]["domain_attention"][0]["name"] = HOSTILE
    hostile["ai"]["topic_attention"][0]["name"] = HOSTILE
    hostile["ai"]["adversaries"][0]["name"] = HOSTILE
    hostile["classified_invocations"][0]["target"] = HOSTILE
    hostile["classified_invocations"][0]["function"] = HOSTILE
    hostile["classified_invocations"][0]["stance"] = HOSTILE
    hostile["distinctive_vocabulary"][0]["term"] = HOSTILE
    hostile["signature_speeches"][0]["title"] = HOSTILE
    hostile["issue_evidence"]["cards"][0]["issue"] = HOSTILE
    hostile["issue_evidence"]["cards"][0]["quote"] = HOSTILE
    hostile["issue_evidence"]["cards"][0]["cite"] = HOSTILE

    payload = compare_site.comparison_payload({"Abraham Lincoln": hostile})
    page = compare_site.render_compare_page(payload)

    assert HOSTILE not in page
    assert html.escape(HOSTILE) in page
    assert r"\u003c/script\u003e" in page
    assert "const escapeHTML = value =>" in page
    assert "option.textContent = P[name].display_name" in page
